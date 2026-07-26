#include "types.h"

#include "shim_common.h"

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

extern "C" {
cudf_status_t cudf_data_type_create(cudf_type_id_t type_id, cudf_data_type_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto* dt = new cudf::data_type(to_cudf_type_id(type_id));
    *out     = static_cast<cudf_data_type_handle>(dt);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_data_type_create_fixed_point(cudf_type_id_t type_id,
                                                int32_t scale,
                                                cudf_data_type_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto* dt = new cudf::data_type(to_cudf_type_id(type_id), scale);
    *out     = static_cast<cudf_data_type_handle>(dt);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_data_type_destroy(cudf_data_type_handle handle)
{
  try {
    delete static_cast<cudf::data_type*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_status_t cudf_data_type_id(cudf_data_type_handle handle, cudf_type_id_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = CUDF_TYPE_EMPTY;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* dt = static_cast<cudf::data_type*>(handle);
    *out     = static_cast<cudf_type_id_t>(dt->id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_data_type_scale(cudf_data_type_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* dt = static_cast<cudf::data_type*>(handle);
    *out     = dt->scale();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_type_size(cudf_type_id_t type_id, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    if (cudf::is_fixed_width(dtype)) {
      *out = static_cast<int32_t>(cudf::size_of(dtype));

    } else {
      *out = 0;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
