#pragma once

#include "execution_context_internal.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace cudf_rust::sorting {

// The `k`-th value of a FLOAT32 or FLOAT64 column's non-null rows under a
// one-key cuDF sort in `order`: NaNs equal each other and sort after every
// other value. Requires 0 < k <= the non-null row count. Null rows may be
// present and are never selected. Returns a valid scalar of the column's type;
// a NaN result means the k-th row is a NaN row. Runs on `exec.stream` and
// allocates the k selected keys and the partial selection's workspace from
// `*exec.mr`; no step sorts every row.
std::unique_ptr<cudf::scalar>
float_top_k_boundary(cudf::column_view const &input, cudf::size_type k,
                     cudf::order order, cudf_exec_context const &exec);

} // namespace cudf_rust::sorting
