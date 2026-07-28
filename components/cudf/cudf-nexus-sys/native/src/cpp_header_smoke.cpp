#include "cudf_rust/cudf_rust.h"

#include <type_traits>

static_assert(!std::is_same_v<cudf_column_handle, cudf_table_handle>);
static_assert(!std::is_same_v<cudf_column_handle, cudf_scalar_handle>);
static_assert(!std::is_same_v<cudf_stream_handle, cudf_event_handle>);
static_assert(!std::is_same_v<cudf_memory_resource_handle, cudf_memory_manager_handle>);
