#include "arrow_device.h"

#include "event.h"
#include "execution_context_internal.hpp"
#include "owning_handle.hpp"
#include "shim_common.h"

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime_api.h>

#include <exception>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

struct ArrowSchema {
  char const *format;
  char const *name;
  char const *metadata;
  int64_t flags;
  int64_t n_children;
  ArrowSchema **children;
  ArrowSchema *dictionary;

  void (*release)(ArrowSchema *);

  void *private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  void const **buffers;
  ArrowArray **children;
  ArrowArray *dictionary;

  void (*release)(ArrowArray *);

  void *private_data;
};

struct ArrowDeviceArray {
  ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  void *sync_event;
  int64_t reserved[3]; // NOLINT(modernize-avoid-c-arrays)
};

void move_arrow_schema(ArrowSchema *src, ArrowSchema *dst) {
  *dst = *src;
  src->release = nullptr;
}

void move_arrow_array(ArrowArray *src, ArrowArray *dst) {
  *dst = *src;
  src->release = nullptr;
}

void move_arrow_device_array(ArrowDeviceArray *src, ArrowDeviceArray *dst) {
  move_arrow_array(&src->array, &dst->array);
  dst->device_id = src->device_id;
  dst->device_type = src->device_type;
  dst->sync_event = src->sync_event;
  for (std::size_t i = 0; i < std::size(dst->reserved); ++i) {
    dst->reserved[i] = src->reserved[i];
  }

  src->device_id = 0;
  src->device_type = 0;
  src->sync_event = nullptr;
  for (auto &value : src->reserved) {
    value = 0;
  }
}

namespace {
constexpr int fixed_width_data_buffer_index = 1;

void destroy_arrow_schema(ArrowSchema *schema) {
  if (schema == nullptr) {
    return;
  }
  if (schema->release != nullptr) {
    schema->release(schema);
  }
  delete schema;
}

void destroy_arrow_device_array(ArrowDeviceArray *array) {
  if (array == nullptr) {
    return;
  }
  if (array->array.release != nullptr) {
    array->array.release(&array->array);
  }
  if (array->array.release != nullptr) {
    std::terminate();
  }
  delete array;
}

using schema_owner_t = std::unique_ptr<ArrowSchema, void (*)(ArrowSchema *)>;
using array_owner_t =
    std::unique_ptr<ArrowDeviceArray, void (*)(ArrowDeviceArray *)>;

struct arrow_device_array_release_state {
  void (*release)(ArrowArray *) = nullptr;
  void *private_data = nullptr;
  cudf_handle_provenance provenance;
};

void release_arrow_device_array_under_provenance(ArrowArray *array) {
  if (array == nullptr) {
    return;
  }
  auto *state =
      static_cast<arrow_device_array_release_state *>(array->private_data);
  if (state == nullptr) {
    return;
  }

  auto const release = state->release;
  auto const private_data = state->private_data;
  auto const provenance = state->provenance;
  array->release = release;
  array->private_data = private_data;

  auto cleanup = cudf_provenance_cleanup_result::completed;
  if (release != nullptr) {
    cleanup = cudf_destroy_under_provenance(
        provenance, [release, array]() { release(array); });
  }

  if (cleanup == cudf_provenance_cleanup_result::not_run) {
    array->release = release_arrow_device_array_under_provenance;
    array->private_data = state;
    return;
  }

  delete state;

  if (cleanup == cudf_provenance_cleanup_result::failed &&
      array->release != nullptr) {
    std::terminate();
  }
}

void wrap_arrow_device_array_release(ArrowDeviceArray *array,
                                     cudf_handle_provenance provenance) {
  if (array == nullptr || array->array.release == nullptr) {
    return;
  }
  auto state = std::make_unique<arrow_device_array_release_state>();
  state->release = array->array.release;
  state->private_data = array->array.private_data;
  state->provenance = std::move(provenance);
  array->array.release = release_arrow_device_array_under_provenance;
  array->array.private_data = state.release();
}

bool arrow_device_column_supported(cudf::column_view const &view) {
  auto const type = view.type();
  if (cudf::is_fixed_width(type) || type.id() == cudf::type_id::STRING) {
    return true;
  }

  if (type.id() == cudf::type_id::LIST) {
    auto const child = cudf::lists_column_view(view).get_sliced_child(
        cudf::get_default_stream());
    return arrow_device_column_supported(child);
  }

  if (type.id() == cudf::type_id::STRUCT) {
    auto const structs_view = cudf::structs_column_view(view);
    for (auto index = 0; index < view.num_children(); ++index) {
      if (!arrow_device_column_supported(structs_view.get_sliced_child(
              index, cudf::get_default_stream()))) {
        return false;
      }
    }
    return true;
  }

  if (type.id() == cudf::type_id::DICTIONARY32) {
    auto const dict_view = cudf::dictionary_column_view(view);
    return dict_view.indices().type().id() == cudf::type_id::INT32 &&
           arrow_device_column_supported(dict_view.keys());
  }

  return false;
}

bool arrow_device_top_level_supported(cudf::table_view const &view) {
  for (auto index = 0; index < view.num_columns(); ++index) {
    if (!arrow_device_column_supported(view.column(index))) {
      return false;
    }
  }
  return true;
}

cudf_arrow_device_export_materialization_t
classify_export_materialization(cudf::column_view const &view) {
  auto const type = view.type();
  if (!arrow_device_column_supported(view)) {
    throw cudf::data_type_error(
        "unsupported cudf type for Arrow device export");
  }
  if (type.id() == cudf::type_id::BOOL8) {
    return CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY;
  }
  if (type.id() == cudf::type_id::STRING && view.size() == 0) {
    // Arrow string arrays must carry a length + 1 offsets buffer, so libcudf
    // materializes a device-side zero scalar for empty exports.
    return CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY;
  }

  if (type.id() == cudf::type_id::LIST) {
    return classify_export_materialization(
        cudf::lists_column_view(view).get_sliced_child(
            cudf::get_default_stream()));
  }

  if (type.id() == cudf::type_id::STRUCT) {
    auto const structs_view = cudf::structs_column_view(view);
    for (auto index = 0; index < view.num_children(); ++index) {
      if (classify_export_materialization(structs_view.get_sliced_child(
              index, cudf::get_default_stream())) ==
          CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY) {
        return CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY;
      }
    }
    return CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
  }

  if (type.id() == cudf::type_id::DICTIONARY32) {
    return classify_export_materialization(
        cudf::dictionary_column_view(view).keys());
  }

  return CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
}

cudf_arrow_device_export_materialization_t
classify_export_materialization(cudf::table_view const &view) {
  if (!arrow_device_top_level_supported(view)) {
    throw cudf::data_type_error(
        "unsupported cudf table type for Arrow device export");
  }

  for (auto index = 0; index < view.num_columns(); ++index) {
    if (classify_export_materialization(view.column(index)) ==
        CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY) {
      return CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY;
    }
  }
  return CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
}

schema_owner_t export_column_schema(cudf::column_view const &view) {
  auto metadata = cudf::interop::get_column_metadata(view);
  auto table_schema = cudf::to_arrow_schema(
      cudf::table_view{{view}}, std::vector<cudf::column_metadata>{metadata});
  auto out = schema_owner_t(new ArrowSchema{}, destroy_arrow_schema);
  move_arrow_schema(table_schema->children[0], out.get());
  return out;
}

array_owner_t export_column_array(cudf::column_view const &view,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr) {
  auto device_array = cudf::to_arrow_device(view, stream, mr);
  auto out = array_owner_t(new ArrowDeviceArray{}, destroy_arrow_device_array);
  move_arrow_device_array(device_array.get(), out.get());
  return out;
}

std::vector<cudf::column_metadata>
export_table_metadata(cudf::table_view const &view,
                      char const *const *column_names,
                      int32_t num_column_names) {
  CUDF_EXPECTS(num_column_names >= 0, "num_column_names must be >= 0",
               std::invalid_argument);
  auto metadata = cudf::interop::get_table_metadata(view);
  if (num_column_names == 0) {
    return metadata;
  }

  CUDF_EXPECTS(column_names != nullptr, "column_names is null",
               std::invalid_argument);
  CUDF_EXPECTS(num_column_names == view.num_columns(),
               "number of column names must equal number of table columns",
               std::invalid_argument);

  for (int32_t index = 0; index < num_column_names; ++index) {
    metadata[static_cast<std::size_t>(index)].name =
        column_names[index] == nullptr ? "" : column_names[index];
  }
  return metadata;
}

schema_owner_t export_table_schema(cudf::table_view const &view,
                                   char const *const *column_names,
                                   int32_t num_column_names) {
  auto metadata = export_table_metadata(view, column_names, num_column_names);
  auto table_schema = cudf::to_arrow_schema(view, metadata);
  auto out = schema_owner_t(new ArrowSchema{}, destroy_arrow_schema);
  move_arrow_schema(table_schema.get(), out.get());
  return out;
}

array_owner_t export_table_array(cudf::table_view const &view,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr) {
  auto device_array = cudf::to_arrow_device(view, stream, mr);
  auto out = array_owner_t(new ArrowDeviceArray{}, destroy_arrow_device_array);
  move_arrow_device_array(device_array.get(), out.get());
  return out;
}

std::size_t first_slice_payload_size(cudf::column_view const &view) {
  auto const null_mask_bytes =
      view.nullable()
          ? static_cast<std::size_t>(cudf::num_bitmask_words(view.size())) *
                sizeof(cudf::bitmask_type)
          : 0;

  if (view.type().id() == cudf::type_id::STRING) {
    std::size_t total = 0;
    if (view.size() > 0) {
      total +=
          static_cast<std::size_t>(view.size() + 1) * sizeof(cudf::size_type);
    }
    total += static_cast<std::size_t>(
        cudf::strings_column_view(view).chars_size(cudf::get_default_stream()));
    return total + null_mask_bytes;
  }

  if (view.type().id() == cudf::type_id::LIST) {
    auto const child = cudf::lists_column_view(view).get_sliced_child(
        cudf::get_default_stream());
    return null_mask_bytes +
           (static_cast<std::size_t>(view.size()) + 1) *
               sizeof(cudf::size_type) +
           first_slice_payload_size(child);
  }

  if (view.type().id() == cudf::type_id::STRUCT) {
    auto const structs_view = cudf::structs_column_view(view);
    auto total = null_mask_bytes;
    for (auto index = 0; index < view.num_children(); ++index) {
      total += first_slice_payload_size(
          structs_view.get_sliced_child(index, cudf::get_default_stream()));
    }
    return total;
  }

  if (view.type().id() == cudf::type_id::DICTIONARY32) {
    auto const dict_view = cudf::dictionary_column_view(view);
    return null_mask_bytes +
           static_cast<std::size_t>(view.size()) * sizeof(int32_t) +
           first_slice_payload_size(dict_view.keys());
  }

  return null_mask_bytes +
         static_cast<std::size_t>(view.size()) * cudf::size_of(view.type());
}

std::size_t first_slice_payload_size(cudf::table_view const &view) {
  std::size_t total = 0;
  for (auto index = 0; index < view.num_columns(); ++index) {
    total += first_slice_payload_size(view.column(index));
  }
  return total;
}

class arrow_device_imported_column_handle_state final
    : public cudf_column_handle_state {
public:
  arrow_device_imported_column_handle_state(cudf::unique_column_view_t view,
                                            cudf_handle_provenance provenance,
                                            bool owns_device_memory)
      : _view(std::make_unique<cudf::unique_column_view_t>(std::move(view))),
        _provenance(std::move(provenance)),
        _owns_device_memory(owns_device_memory) {}

  ~arrow_device_imported_column_handle_state() override {
    if (!_owns_device_memory) {
      return;
    }
    auto view = std::move(_view);
    auto const cleanup =
        cudf_destroy_under_provenance(_provenance, [&view]() { view.reset(); });
    if (cleanup != cudf_provenance_cleanup_result::completed &&
        view != nullptr) {
      std::terminate();
    }
  }

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override {
    return view_ref().size();
  }

  [[nodiscard]] cudf::column_view view() const override { return view_ref(); }

  [[nodiscard]] std::unique_ptr<cudf::column>
  deep_copy(rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view_ref(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view) const override {
    return first_slice_payload_size(view_ref());
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return first_slice_payload_size(view_ref());
  }

private:
  [[nodiscard]] cudf::column_view const &view_ref() const { return **_view; }

  std::unique_ptr<cudf::unique_column_view_t> _view;
  cudf_handle_provenance _provenance;
  bool _owns_device_memory;
};

struct arrow_device_imported_table_root_state {
  arrow_device_imported_table_root_state(cudf::unique_table_view_t imported,
                                         cudf_handle_provenance provenance,
                                         bool owns_device_memory)
      : view(std::make_unique<cudf::unique_table_view_t>(std::move(imported))),
        provenance(std::move(provenance)),
        owns_device_memory(owns_device_memory) {}

  ~arrow_device_imported_table_root_state() {
    if (!owns_device_memory) {
      return;
    }
    auto local_view = std::move(view);
    auto const cleanup = cudf_destroy_under_provenance(
        provenance, [&local_view]() { local_view.reset(); });
    if (cleanup != cudf_provenance_cleanup_result::completed &&
        local_view != nullptr) {
      std::terminate();
    }
  }

  [[nodiscard]] cudf::table_view const &table_view_ref() const {
    return **view;
  }

  std::unique_ptr<cudf::unique_table_view_t> view;
  cudf_handle_provenance provenance;
  bool owns_device_memory;
};

class arrow_device_imported_table_column_handle_state final
    : public cudf_column_handle_state {
public:
  arrow_device_imported_table_column_handle_state(
      std::shared_ptr<arrow_device_imported_table_root_state> root,
      std::size_t index)
      : _root(std::move(root)), _index(index) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type size() const override { return view().size(); }

  [[nodiscard]] cudf::column_view view() const override {
    return _root->table_view_ref().column(static_cast<cudf::size_type>(_index));
  }

  [[nodiscard]] std::unique_ptr<cudf::column>
  deep_copy(rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view) const override {
    return first_slice_payload_size(view());
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return first_slice_payload_size(view());
  }

private:
  std::shared_ptr<arrow_device_imported_table_root_state> _root;
  std::size_t _index;
};

class arrow_device_imported_table_handle_state final
    : public cudf_table_handle_state {
public:
  explicit arrow_device_imported_table_handle_state(
      std::shared_ptr<arrow_device_imported_table_root_state> root)
      : _root(std::move(root)) {}

  [[nodiscard]] bool is_native() const noexcept override { return false; }

  [[nodiscard]] cudf::size_type num_columns() const override {
    return _root->table_view_ref().num_columns();
  }

  [[nodiscard]] cudf::size_type num_rows() const override {
    return _root->table_view_ref().num_rows();
  }

  [[nodiscard]] cudf::table_view view() const override {
    return _root->table_view_ref();
  }

  [[nodiscard]] std::unique_ptr<cudf::table>
  deep_copy(rmm::cuda_stream_view stream,
            rmm::device_async_resource_ref mr) const override {
    return std::make_unique<cudf::table>(_root->table_view_ref(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view) const override {
    return first_slice_payload_size(_root->table_view_ref());
  }

  [[nodiscard]] std::size_t device_allocated_size() const override {
    return first_slice_payload_size(_root->table_view_ref());
  }

  std::vector<std::unique_ptr<cudf_column_handle_state>>
  release_columns() override {
    std::vector<std::unique_ptr<cudf_column_handle_state>> columns;
    auto const num_columns = _root->table_view_ref().num_columns();
    columns.reserve(static_cast<std::size_t>(num_columns));
    for (auto index = 0; index < num_columns; ++index) {
      columns.push_back(
          std::make_unique<arrow_device_imported_table_column_handle_state>(
              _root, index));
    }
    return columns;
  }

  void restore_released_columns(
      cudf::size_type,
      std::vector<std::unique_ptr<cudf_column_handle_state>>) override {
    // Releasing imported Arrow-device columns does not mutate the table root.
  }

private:
  std::shared_ptr<arrow_device_imported_table_root_state> _root;
};

ArrowDeviceArray const *
require_arrow_device_array(cudf_arrow_device_array_handle handle) {
  CUDF_EXPECTS(handle != nullptr, "ArrowDeviceArray handle is null",
               std::invalid_argument);
  return static_cast<ArrowDeviceArray const *>(handle);
}

ArrowSchema const *require_arrow_schema(cudf_arrow_schema_handle handle) {
  CUDF_EXPECTS(handle != nullptr, "ArrowSchema handle is null",
               std::invalid_argument);
  return static_cast<ArrowSchema const *>(handle);
}

ArrowSchema const *require_table_child_schema(ArrowSchema const *schema,
                                              int64_t index) {
  CUDF_EXPECTS(schema->n_children >= 0,
               "Arrow table schema child count must be >= 0",
               std::invalid_argument);
  CUDF_EXPECTS(index >= 0 && index < schema->n_children,
               "Arrow table schema child index is out of bounds",
               std::invalid_argument);
  CUDF_EXPECTS(schema->children != nullptr,
               "Arrow table schema children pointer is null",
               std::invalid_argument);
  CUDF_EXPECTS(schema->children[index] != nullptr,
               "Arrow table schema child is null", std::invalid_argument);
  return schema->children[index];
}

bool arrow_schema_is_supported_fixed_width_leaf(ArrowSchema const *schema) {
  CUDF_EXPECTS(schema != nullptr, "Arrow schema is null",
               std::invalid_argument);
  CUDF_EXPECTS(schema->format != nullptr, "Arrow schema format is null",
               std::invalid_argument);
  CUDF_EXPECTS(schema->n_children >= 0, "Arrow schema child count must be >= 0",
               std::invalid_argument);

  auto const format = std::string_view{schema->format};
  if (format.empty()) {
    return false;
  }

  if (schema->dictionary != nullptr || schema->n_children != 0) {
    return false;
  }

  switch (format.front()) {
  case 'c':
  case 'C':
  case 's':
  case 'S':
  case 'i':
  case 'I':
  case 'l':
  case 'L':
  case 'e':
  case 'f':
  case 'g':
  case 'w':
    return true;
  case 'd':
    return format.starts_with("d:");
  case 't':
    return format.starts_with("td") || format.starts_with("ts") ||
           format.starts_with("tt") || format.starts_with("tD");
  default:
    return false;
  }
}

void validate_supported_fixed_width_leaf_schema(ArrowSchema const *schema,
                                                char const *context) {
  if (arrow_schema_is_supported_fixed_width_leaf(schema)) {
    return;
  }
  throw std::invalid_argument(context);
}

ArrowArray const *require_table_child_array(ArrowDeviceArray const *array,
                                            int64_t index) {
  CUDF_EXPECTS(array->array.n_children >= 0,
               "Arrow table array child count must be >= 0",
               std::invalid_argument);
  CUDF_EXPECTS(index >= 0 && index < array->array.n_children,
               "Arrow table array child index is out of bounds",
               std::invalid_argument);
  CUDF_EXPECTS(array->array.children != nullptr,
               "Arrow table array children pointer is null",
               std::invalid_argument);
  CUDF_EXPECTS(array->array.children[index] != nullptr,
               "Arrow table array child is null", std::invalid_argument);
  return array->array.children[index];
}

cudf_status_t
observed_wait_on_sync_event_if_present(ArrowDeviceArray *array,
                                       rmm::cuda_stream_view stream) {
  if (array->sync_event == nullptr) {
    return CUDF_SUCCESS;
  }

  auto const raw_event = reinterpret_cast<void *>(
      *reinterpret_cast<cudaEvent_t *>(array->sync_event));
  CUDF_EXPECTS(raw_event != nullptr,
               "ArrowDeviceArray sync_event points to a null CUDA event",
               std::invalid_argument);

  auto const wait_status =
      cudf_raw_cuda_stream_wait_event(stream.value(), raw_event);
  if (wait_status != CUDF_SUCCESS) {
    return wait_status;
  }

  array->sync_event = nullptr;
  return CUDF_SUCCESS;
}

cudf_status_t
observed_record_sync_event_if_present(ArrowDeviceArray const *array,
                                      rmm::cuda_stream_view stream) {
  if (array->sync_event == nullptr) {
    return CUDF_SUCCESS;
  }

  auto const raw_stream = reinterpret_cast<void *>(stream.value());
  if (raw_stream == nullptr) {
    // The raw observation hook currently requires a non-null stream pointer, so
    // default-stream exports keep their existing sync contract without
    // contributing explicit observation data.
    return CUDF_SUCCESS;
  }

  auto const raw_event = reinterpret_cast<void *>(
      *reinterpret_cast<cudaEvent_t *>(array->sync_event));
  CUDF_EXPECTS(raw_event != nullptr,
               "ArrowDeviceArray sync_event points to a null CUDA event",
               std::invalid_argument);

  return cudf_raw_cuda_event_record(raw_event, raw_stream);
}
} // namespace

extern "C" {
cudf_status_t cudf_arrow_device_column_export(
    cudf_column_view_handle view, cudf_exec_options_t const *options,
    cudf_arrow_schema_handle *out_schema,
    cudf_arrow_device_array_handle *out_array,
    cudf_arrow_device_export_materialization_t *out_materialization) {
  CUDF_RUST_NULL_CHECK(out_schema, "out_schema is null");
  CUDF_RUST_NULL_CHECK(out_array, "out_array is null");
  CUDF_RUST_NULL_CHECK(out_materialization, "out_materialization is null");
  *out_schema = nullptr;
  *out_array = nullptr;
  *out_materialization = CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
  CUDF_RUST_NULL_CHECK(view, "view is null");

  CUDF_RUST_TRY {
    auto const &column_view = *static_cast<cudf::column_view *>(view);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }
    {
      *out_materialization = classify_export_materialization(column_view);
      auto schema = export_column_schema(column_view);
      auto array = export_column_array(column_view, exec->stream, *exec->mr);
      cudf_status_t const observe_status =
          observed_record_sync_event_if_present(array.get(), exec->stream);
      if (observe_status != CUDF_SUCCESS) {
        return observe_status;
      }
      wrap_arrow_device_array_release(array.get(),
                                      cudf_capture_exec_provenance(*exec));
      *out_schema = static_cast<cudf_arrow_schema_handle>(schema.release());
      *out_array = static_cast<cudf_arrow_device_array_handle>(array.release());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_arrow_device_table_export(
    cudf_table_view_handle view, cudf_exec_options_t const *options,
    char const *const *column_names, int32_t num_column_names,
    cudf_arrow_schema_handle *out_schema,
    cudf_arrow_device_array_handle *out_array,
    cudf_arrow_device_export_materialization_t *out_materialization) {
  CUDF_RUST_NULL_CHECK(out_schema, "out_schema is null");
  CUDF_RUST_NULL_CHECK(out_array, "out_array is null");
  CUDF_RUST_NULL_CHECK(out_materialization, "out_materialization is null");
  *out_schema = nullptr;
  *out_array = nullptr;
  *out_materialization = CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW;
  CUDF_RUST_NULL_CHECK(view, "view is null");

  CUDF_RUST_TRY {
    auto const &table_view = *static_cast<cudf::table_view *>(view);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }
    {
      *out_materialization = classify_export_materialization(table_view);
      auto schema =
          export_table_schema(table_view, column_names, num_column_names);
      auto array = export_table_array(table_view, exec->stream, *exec->mr);
      cudf_status_t const observe_status =
          observed_record_sync_event_if_present(array.get(), exec->stream);
      if (observe_status != CUDF_SUCCESS) {
        return observe_status;
      }
      wrap_arrow_device_array_release(array.get(),
                                      cudf_capture_exec_provenance(*exec));
      *out_schema = static_cast<cudf_arrow_schema_handle>(schema.release());
      *out_array = static_cast<cudf_arrow_device_array_handle>(array.release());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_arrow_device_column_import(
    cudf_arrow_schema_handle schema, cudf_arrow_device_array_handle array,
    cudf_exec_options_t const *options, cudf_column_handle *out,
    cudf_arrow_device_import_materialization_t *out_materialization) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_NULL_CHECK(out_materialization, "out_materialization is null");
  *out = nullptr;
  *out_materialization = CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;

  CUDF_RUST_TRY {
    auto const *schema_ptr = require_arrow_schema(schema);
    auto const *array_ptr = require_arrow_device_array(array);

    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto staged_input = *array_ptr;
    cudf_status_t const wait_status =
        observed_wait_on_sync_event_if_present(&staged_input, exec->stream);
    if (wait_status != CUDF_SUCCESS) {
      return wait_status;
    }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto imported = cudf::from_arrow_device_column(schema_ptr, &staged_input,
                                                     exec->stream, *exec->mr);
      if (!arrow_device_column_supported(*imported)) {
        throw cudf::data_type_error(
            "unsupported cudf type for Arrow device import");
      }
      bool const owns_device_memory =
          !imported.get_deleter().owned_mem_.empty();
      *out_materialization =
          owns_device_memory ? CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY
                             : CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;
      auto state = std::make_unique<arrow_device_imported_column_handle_state>(
          std::move(imported), cudf_capture_exec_provenance(*exec),
          owns_device_memory);
      *out = cudf_release_column_handle(std::move(state));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_arrow_device_table_import(
    cudf_arrow_schema_handle schema, cudf_arrow_device_array_handle array,
    cudf_exec_options_t const *options, cudf_table_handle *out,
    cudf_arrow_device_import_materialization_t *out_materialization) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_NULL_CHECK(out_materialization, "out_materialization is null");
  *out = nullptr;
  *out_materialization = CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;

  CUDF_RUST_TRY {
    auto const *schema_ptr = require_arrow_schema(schema);
    auto const *array_ptr = require_arrow_device_array(array);

    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto staged_input = *array_ptr;
    cudf_status_t const wait_status =
        observed_wait_on_sync_event_if_present(&staged_input, exec->stream);
    if (wait_status != CUDF_SUCCESS) {
      return wait_status;
    }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto imported = cudf::from_arrow_device(schema_ptr, &staged_input,
                                              exec->stream, *exec->mr);
      if (!arrow_device_top_level_supported(*imported)) {
        throw cudf::data_type_error(
            "unsupported cudf table type for Arrow device import");
      }
      bool const owns_device_memory =
          !imported.get_deleter().owned_mem_.empty();
      *out_materialization =
          owns_device_memory ? CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY
                             : CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW;
      auto root = std::make_shared<arrow_device_imported_table_root_state>(
          std::move(imported), cudf_capture_exec_provenance(*exec),
          owns_device_memory);
      auto state = std::make_unique<arrow_device_imported_table_handle_state>(
          std::move(root));
      *out = cudf_release_table_handle(std::move(state));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_arrow_device_schema_destroy(cudf_arrow_schema_handle schema) {
  destroy_arrow_schema(static_cast<ArrowSchema *>(schema));
}

void cudf_arrow_device_array_destroy(cudf_arrow_device_array_handle array) {
  destroy_arrow_device_array(static_cast<ArrowDeviceArray *>(array));
}

cudf_status_t
cudf_arrow_device_array_device_id(cudf_arrow_device_array_handle array,
                                  int64_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = -1;
  CUDF_RUST_TRY {
    *out = require_arrow_device_array(array)->device_id;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_arrow_device_array_device_type(cudf_arrow_device_array_handle array,
                                    int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = -1;
  CUDF_RUST_TRY {
    *out = require_arrow_device_array(array)->device_type;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_test_arrow_device_array_set_device_type(
    cudf_arrow_device_array_handle array, int32_t device_type) {
  if (array == nullptr) {
    return;
  }
  static_cast<ArrowDeviceArray *>(array)->device_type = device_type;
}

cudf_status_t
cudf_arrow_device_array_has_sync_event(cudf_arrow_device_array_handle array,
                                       int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY {
    *out = require_arrow_device_array(array)->sync_event != nullptr ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_arrow_device_array_sync_event_raw(cudf_arrow_device_array_handle array,
                                       void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    *out = require_arrow_device_array(array)->sync_event;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_arrow_device_column_fixed_width_values_ptr(
    cudf_arrow_schema_handle schema, cudf_arrow_device_array_handle array,
    void const **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;

  CUDF_RUST_TRY {
    auto const *schema_ptr = require_arrow_schema(schema);
    auto const *array_ptr = require_arrow_device_array(array);
    validate_supported_fixed_width_leaf_schema(
        schema_ptr, "Arrow schema is not a supported fixed-width leaf for "
                    "values-buffer access");
    CUDF_EXPECTS(array_ptr->array.n_buffers > fixed_width_data_buffer_index,
                 "ArrowDeviceArray is missing the fixed-width values buffer",
                 std::invalid_argument);
    *out = array_ptr->array.buffers[fixed_width_data_buffer_index];
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_arrow_device_table_num_columns(cudf_arrow_schema_handle schema,
                                    int64_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY {
    *out = require_arrow_schema(schema)->n_children;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_arrow_device_table_column_name(cudf_arrow_schema_handle schema,
                                    int64_t index, char const **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto const *child =
        require_table_child_schema(require_arrow_schema(schema), index);
    *out = child->name;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_arrow_device_table_fixed_width_values_ptr(
    cudf_arrow_schema_handle schema, cudf_arrow_device_array_handle array,
    int64_t index, void const **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;

  CUDF_RUST_TRY {
    auto const *child_schema =
        require_table_child_schema(require_arrow_schema(schema), index);
    auto const *child_array =
        require_table_child_array(require_arrow_device_array(array), index);
    validate_supported_fixed_width_leaf_schema(
        child_schema, "Arrow table column is not a supported fixed-width leaf "
                      "for values-buffer access");
    CUDF_EXPECTS(
        child_array->n_buffers > fixed_width_data_buffer_index,
        "ArrowDeviceArray child is missing the fixed-width values buffer",
        std::invalid_argument);
    *out = child_array->buffers[fixed_width_data_buffer_index];
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
