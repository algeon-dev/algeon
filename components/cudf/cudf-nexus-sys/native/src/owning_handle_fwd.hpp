#ifndef CUDF_RUST_OWNING_HANDLE_FWD_HPP
#define CUDF_RUST_OWNING_HANDLE_FWD_HPP

#include "cudf_rust/column.h"
#include "cudf_rust/table.h"
#include "handle_provenance.hpp"

#include <memory>

namespace cudf {
class column;
class table;
}  // namespace cudf

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column);

cudf_column_handle cudf_release_column_handle(std::unique_ptr<cudf::column> column,
                                              cudf_handle_provenance provenance);

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table);

cudf_table_handle cudf_release_table_handle(std::unique_ptr<cudf::table> table,
                                            cudf_handle_provenance provenance);

#endif  // CUDF_RUST_OWNING_HANDLE_FWD_HPP
