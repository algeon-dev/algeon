#ifndef CUDF_RUST_OWNING_HANDLE_FWD_HPP
#define CUDF_RUST_OWNING_HANDLE_FWD_HPP

#include "column.h"
#include "table.h"

#include <memory>

namespace cudf {
class column;
class table;
}  // namespace cudf

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column);
cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table);

#endif  // CUDF_RUST_OWNING_HANDLE_FWD_HPP
