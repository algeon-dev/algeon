#ifndef CUDF_RUST_OWNING_HANDLE_HPP
#define CUDF_RUST_OWNING_HANDLE_HPP

#include "column.h"
#include "handle_provenance.hpp"
#include "table.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <optional>
#include <vector>

struct cudf_column_handle_state {
  virtual ~cudf_column_handle_state() = default;

  [[nodiscard]] virtual bool is_native() const noexcept = 0;

  [[nodiscard]] virtual cudf::size_type size() const = 0;

  [[nodiscard]] virtual cudf::column_view view() const = 0;

  [[nodiscard]] virtual std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

  [[nodiscard]] virtual std::size_t payload_size(rmm::cuda_stream_view stream) const = 0;

  [[nodiscard]] virtual std::size_t device_allocated_size() const = 0;

  [[nodiscard]] virtual std::optional<cudf_handle_provenance> destroy_provenance() const noexcept;

  virtual std::unique_ptr<cudf::column> release_native_column();
};

struct cudf_table_handle_state {
  virtual ~cudf_table_handle_state() = default;

  [[nodiscard]] virtual bool is_native() const noexcept = 0;

  [[nodiscard]] virtual cudf::size_type num_columns() const = 0;

  [[nodiscard]] virtual cudf::size_type num_rows() const = 0;

  [[nodiscard]] virtual cudf::table_view view() const = 0;

  [[nodiscard]] virtual std::unique_ptr<cudf::table> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

  [[nodiscard]] virtual std::size_t payload_size(rmm::cuda_stream_view stream) const = 0;

  [[nodiscard]] virtual std::size_t device_allocated_size() const = 0;

  [[nodiscard]] virtual std::optional<cudf_handle_provenance> destroy_provenance() const noexcept;

  virtual std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns();

  virtual void restore_released_columns(
    cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns);
};

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column);

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column,
                                              cudf_handle_provenance provenance);

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf_column_handle_state> column);

cudf_column_handle cudf_publish_column_handle(std::unique_ptr<cudf_column_handle_state>& column);

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table);

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table,
                                            cudf_handle_provenance provenance);

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf_table_handle_state> table);

cudf_status_t cudf_destroy_column_handle_under_provenance(cudf_column_handle handle) noexcept;

cudf_status_t cudf_destroy_table_handle_under_provenance(cudf_table_handle handle) noexcept;

cudf_column_handle_state& cudf_column_state_from_handle(cudf_column_handle handle);

cudf_table_handle_state& cudf_table_state_from_handle(cudf_table_handle handle);

std::unique_ptr<cudf_column_handle_state> cudf_take_column_state(cudf_column_handle handle);

std::unique_ptr<cudf_table_handle_state> cudf_take_table_state(cudf_table_handle handle);

std::unique_ptr<cudf_column_handle_state> cudf_make_foreign_column_handle_state(
  uint32_t root_column_id,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings);

std::unique_ptr<cudf_table_handle_state> cudf_make_foreign_table_handle_state(
  cudf_interop_semantic_table_t const* table,
  cudf_interop_semantic_column_t const* semantic_columns,
  int32_t num_semantic_columns,
  cudf_interop_buffer_binding_t const* bindings,
  int32_t num_bindings);

std::unique_ptr<cudf_table_handle_state> cudf_make_table_handle_state_from_columns(
  cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns);

std::unique_ptr<cudf_table_handle_state> cudf_make_foreign_table_handle_state_from_columns(
  cudf::size_type num_rows, std::vector<std::unique_ptr<cudf_column_handle_state>> columns);

#endif  // CUDF_RUST_OWNING_HANDLE_HPP
