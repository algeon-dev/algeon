#include "cuml_rust/cuml_rust.h"

static int cuml_rust_c_header_smoke(void)
{
  cuml_handle_t handle = (cuml_handle_t)0;
  cuml_error_detail_t const* detail = cuml_last_error_detail();
  return handle != (cuml_handle_t)0 || detail != (cuml_error_detail_t const*)0;
}
